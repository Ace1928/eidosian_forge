import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
class ChatKinetica(BaseChatModel):
    """Kinetica LLM Chat Model API.

    Prerequisites for using this API:

    * The ``gpudb`` and ``typeguard`` packages installed.
    * A Kinetica DB instance.
    * Kinetica host specified in ``KINETICA_URL``
    * Kinetica login specified ``KINETICA_USER``, and ``KINETICA_PASSWD``.
    * An LLM context that specifies the tables and samples to use for inferencing.

    This API is intended to interact with the Kinetica SqlAssist LLM that supports
    generation of SQL from natural language.

    In the Kinetica LLM workflow you create an LLM context in the database that provides
    information needed for infefencing that includes tables, annotations, rules, and
    samples. Invoking ``load_messages_from_context()`` will retrieve the contxt
    information from the database so that it can be used to create a chat prompt.

    The chat prompt consists of a ``SystemMessage`` and pairs of
    ``HumanMessage``/``AIMessage`` that contain the samples which are question/SQL
    pairs. You can append pairs samples to this list but it is not intended to
    facilitate a typical natural language conversation.

    When you create a chain from the chat prompt and execute it, the Kinetica LLM will
    generate SQL from the input. Optionally you can use ``KineticaSqlOutputParser`` to
    execute the SQL and return the result as a dataframe.

    The following example creates an LLM using the environment variables for the
    Kinetica connection. This will fail if the API is unable to connect to the database.

    Example:
        .. code-block:: python

            from langchain_community.chat_models.kinetica import KineticaChatLLM
            kinetica_llm = KineticaChatLLM()

    If you prefer to pass connection information directly then you can create a
    connection using ``KineticaUtil.create_kdbc()``.

    Example:
        .. code-block:: python

            from langchain_community.chat_models.kinetica import (
                KineticaChatLLM, KineticaUtil)
            kdbc = KineticaUtil._create_kdbc(url=url, user=user, passwd=passwd)
            kinetica_llm = KineticaChatLLM(kdbc=kdbc)
    """
    kdbc: Any = Field(exclude=True)
    ' Kinetica DB connection. '

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Pydantic object validator."""
        kdbc = values.get('kdbc', None)
        if kdbc is None:
            kdbc = KineticaUtil.create_kdbc()
            values['kdbc'] = kdbc
        return values

    @property
    def _llm_type(self) -> str:
        return 'kinetica-sqlassist'

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return dict(kinetica_version=str(self.kdbc.server_version), api_version=version('gpudb'))

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        if stop is not None:
            raise ValueError('stop kwargs are not permitted.')
        dict_messages = [self._convert_message_to_dict(m) for m in messages]
        sql_response = self._submit_completion(dict_messages)
        response_message = sql_response.choices[0].message
        generated_dict = response_message.dict()
        generated_message = self._convert_message_from_dict(generated_dict)
        llm_output = dict(input_tokens=sql_response.usage.prompt_tokens, output_tokens=sql_response.usage.completion_tokens, model_name=sql_response.model)
        return ChatResult(generations=[ChatGeneration(message=generated_message)], llm_output=llm_output)

    def load_messages_from_context(self, context_name: str) -> List:
        """Load a lanchain prompt from a Kinetica context.

        A Kinetica Context is an object created with the Kinetica Workbench UI or with
        SQL syntax. This function will convert the data in the context to a list of
        messages that can be used as a prompt. The messages will contain a
        ``SystemMessage`` followed by pairs of ``HumanMessage``/``AIMessage`` that
        contain the samples.

        Args:
            context_name: The name of an LLM context in the database.

        Returns:
            A list of messages containing the information from the context.
        """
        sql = f"GENERATE PROMPT WITH OPTIONS (CONTEXT_NAMES = '{context_name}')"
        result = self._execute_sql(sql)
        prompt = result['Prompt']
        prompt_json = json.loads(prompt)
        request = _KdtoSuggestRequest.parse_obj(prompt_json)
        payload = request.payload
        dict_messages = []
        dict_messages.append(dict(role='system', content=payload.get_system_str()))
        dict_messages.extend(payload.get_messages())
        messages = [self._convert_message_from_dict(m) for m in dict_messages]
        return messages

    def _submit_completion(self, messages: List[Dict]) -> _KdtSqlResponse:
        """Submit a /chat/completions request to Kinetica."""
        request = dict(messages=messages)
        request_json = json.dumps(request)
        response_raw = self.kdbc._GPUdb__submit_request_json('/chat/completions', request_json)
        response_json = json.loads(response_raw)
        status = response_json['status']
        if status != 'OK':
            message = response_json['message']
            match_resp = re.compile('response:({.*})')
            result = match_resp.search(message)
            if result is not None:
                response = result.group(1)
                response_json = json.loads(response)
                message = response_json['message']
            raise ValueError(message)
        data = response_json['data']
        response = _KdtCompletionResponse.parse_obj(data)
        if response.status != 'OK':
            raise ValueError('SQL Generation failed')
        return response.data

    def _execute_sql(self, sql: str) -> Dict:
        """Execute an SQL query and return the result."""
        response = self.kdbc.execute_sql_and_decode(sql, limit=1, get_column_major=False)
        status_info = response['status_info']
        if status_info['status'] != 'OK':
            message = status_info['message']
            raise ValueError(message)
        records = response['records']
        if len(records) != 1:
            raise ValueError('No records returned.')
        record = records[0]
        response_dict = {}
        for col, val in record.items():
            response_dict[col] = val
        return response_dict

    @classmethod
    def load_messages_from_datafile(cls, sa_datafile: Path) -> List[BaseMessage]:
        """Load a lanchain prompt from a Kinetica context datafile."""
        datafile_dict = _KineticaLlmFileContextParser.parse_dialogue_file(sa_datafile)
        messages = cls._convert_dict_to_messages(datafile_dict)
        return messages

    @classmethod
    def _convert_message_to_dict(cls, message: BaseMessage) -> Dict:
        """Convert a single message to a BaseMessage."""
        content = cast(str, message.content)
        if isinstance(message, HumanMessage):
            role = 'user'
        elif isinstance(message, AIMessage):
            role = 'assistant'
        elif isinstance(message, SystemMessage):
            role = 'system'
        else:
            raise ValueError(f'Got unsupported message type: {message}')
        result_message = dict(role=role, content=content)
        return result_message

    @classmethod
    def _convert_message_from_dict(cls, message: Dict) -> BaseMessage:
        """Convert a single message from a BaseMessage."""
        role = message['role']
        content = message['content']
        if role == 'user':
            return HumanMessage(content=content)
        elif role == 'assistant':
            return AIMessage(content=content)
        elif role == 'system':
            return SystemMessage(content=content)
        else:
            raise ValueError(f'Got unsupported role: {role}')

    @classmethod
    def _convert_dict_to_messages(cls, sa_data: Dict) -> List[BaseMessage]:
        """Convert a dict to a list of BaseMessages."""
        schema = sa_data['schema']
        system = sa_data['system']
        messages = sa_data['messages']
        LOG.info(f'Importing prompt for schema: {schema}')
        result_list: List[BaseMessage] = []
        result_list.append(SystemMessage(content=system))
        result_list.extend([cls._convert_message_from_dict(m) for m in messages])
        return result_list