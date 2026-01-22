from typing import Any, Dict, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.pyfunc.model import (
from mlflow.types.llm import ChatMessage, ChatParams, ChatResponse
from mlflow.utils.annotations import experimental
@experimental
class _ChatModelPyfuncWrapper:
    """
    Wrapper class that converts dict inputs to pydantic objects accepted by :class:`~ChatModel`.
    """

    def __init__(self, chat_model, context, signature):
        """
        Args:
            chat_model: An instance of a subclass of :class:`~ChatModel`.
            context: A :class:`~PythonModelContext` instance containing artifacts that
                        ``chat_model`` may use when performing inference.
            signature: :class:`~ModelSignature` instance describing model input and output.
        """
        self.chat_model = chat_model
        self.context = context
        self.signature = signature

    def _convert_input(self, model_input):
        dict_input = {key: value[0] for key, value in model_input.to_dict(orient='list').items()}
        messages = [ChatMessage(**message) for message in dict_input.pop('messages', [])]
        params = ChatParams(**dict_input)
        return (messages, params)

    def predict(self, model_input: Dict[str, Any], params: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """
        Args:
            model_input: Model input data in the form of a chat request.
            params: Additional parameters to pass to the model for inference.
                       Unused in this implementation, as the params are handled
                       via ``self._convert_input()``.

        Returns:
            Model predictions in :py:class:`~ChatResponse` format.
        """
        messages, params = self._convert_input(model_input)
        response = self.chat_model.predict(self.context, messages, params)
        if not isinstance(response, ChatResponse):
            raise MlflowException(f'Model returned an invalid response. Expected a ChatResponse, but got {type(response)} instead.', error_code=INTERNAL_ERROR)
        return response.to_dict()