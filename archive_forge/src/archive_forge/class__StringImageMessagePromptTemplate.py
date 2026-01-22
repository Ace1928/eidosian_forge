from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from langchain_core._api import deprecated
from langchain_core.load import Serializable
from langchain_core.messages import (
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompt_values import ChatPromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import StringPromptTemplate, get_template_variables
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_colored_text
from langchain_core.utils.interactive_env import is_interactive_env
class _StringImageMessagePromptTemplate(BaseMessagePromptTemplate):
    """Human message prompt template. This is a message sent from the user."""
    prompt: Union[StringPromptTemplate, List[Union[StringPromptTemplate, ImagePromptTemplate]]]
    'Prompt template.'
    additional_kwargs: dict = Field(default_factory=dict)
    'Additional keyword arguments to pass to the prompt template.'
    _msg_class: Type[BaseMessage]

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'prompts', 'chat']

    @classmethod
    def from_template(cls: Type[_StringImageMessagePromptTemplateT], template: Union[str, List[Union[str, _TextTemplateParam, _ImageTemplateParam]]], template_format: str='f-string', **kwargs: Any) -> _StringImageMessagePromptTemplateT:
        """Create a class from a string template.

        Args:
            template: a template.
            template_format: format of the template.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        if isinstance(template, str):
            prompt: Union[StringPromptTemplate, List] = PromptTemplate.from_template(template, template_format=template_format)
            return cls(prompt=prompt, **kwargs)
        elif isinstance(template, list):
            prompt = []
            for tmpl in template:
                if isinstance(tmpl, str) or (isinstance(tmpl, dict) and 'text' in tmpl):
                    if isinstance(tmpl, str):
                        text: str = tmpl
                    else:
                        text = cast(_TextTemplateParam, tmpl)['text']
                    prompt.append(PromptTemplate.from_template(text, template_format=template_format))
                elif isinstance(tmpl, dict) and 'image_url' in tmpl:
                    img_template = cast(_ImageTemplateParam, tmpl)['image_url']
                    if isinstance(img_template, str):
                        vars = get_template_variables(img_template, 'f-string')
                        if vars:
                            if len(vars) > 1:
                                raise ValueError(f'Only one format variable allowed per image template.\nGot: {vars}\nFrom: {tmpl}')
                            input_variables = [vars[0]]
                        else:
                            input_variables = None
                        img_template = {'url': img_template}
                        img_template_obj = ImagePromptTemplate(input_variables=input_variables, template=img_template)
                    elif isinstance(img_template, dict):
                        img_template = dict(img_template)
                        if 'url' in img_template:
                            input_variables = get_template_variables(img_template['url'], 'f-string')
                        else:
                            input_variables = None
                        img_template_obj = ImagePromptTemplate(input_variables=input_variables, template=img_template)
                    else:
                        raise ValueError()
                    prompt.append(img_template_obj)
                else:
                    raise ValueError()
            return cls(prompt=prompt, **kwargs)
        else:
            raise ValueError()

    @classmethod
    def from_template_file(cls: Type[_StringImageMessagePromptTemplateT], template_file: Union[str, Path], input_variables: List[str], **kwargs: Any) -> _StringImageMessagePromptTemplateT:
        """Create a class from a template file.

        Args:
            template_file: path to a template file. String or Path.
            input_variables: list of input variables.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        with open(str(template_file), 'r') as f:
            template = f.read()
        return cls.from_template(template, input_variables=input_variables, **kwargs)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> List[str]:
        """
        Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        input_variables = [iv for prompt in prompts for iv in prompt.input_variables]
        return input_variables

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = self.prompt.format(**kwargs)
            return self._msg_class(content=text, additional_kwargs=self.additional_kwargs)
        else:
            content: List = []
            for prompt in self.prompt:
                inputs = {var: kwargs[var] for var in prompt.input_variables}
                if isinstance(prompt, StringPromptTemplate):
                    formatted: Union[str, ImageURL] = prompt.format(**inputs)
                    content.append({'type': 'text', 'text': formatted})
                elif isinstance(prompt, ImagePromptTemplate):
                    formatted = prompt.format(**inputs)
                    content.append({'type': 'image_url', 'image_url': formatted})
            return self._msg_class(content=content, additional_kwargs=self.additional_kwargs)

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = await self.prompt.aformat(**kwargs)
            return self._msg_class(content=text, additional_kwargs=self.additional_kwargs)
        else:
            content: List = []
            for prompt in self.prompt:
                inputs = {var: kwargs[var] for var in prompt.input_variables}
                if isinstance(prompt, StringPromptTemplate):
                    formatted: Union[str, ImageURL] = await prompt.aformat(**inputs)
                    content.append({'type': 'text', 'text': formatted})
                elif isinstance(prompt, ImagePromptTemplate):
                    formatted = await prompt.aformat(**inputs)
                    content.append({'type': 'image_url', 'image_url': formatted})
            return self._msg_class(content=content, additional_kwargs=self.additional_kwargs)

    def pretty_repr(self, html: bool=False) -> str:
        title = self.__class__.__name__.replace('MessagePromptTemplate', ' Message')
        title = get_msg_title_repr(title, bold=html)
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        prompt_reprs = '\n\n'.join((prompt.pretty_repr(html=html) for prompt in prompts))
        return f'{title}\n\n{prompt_reprs}'