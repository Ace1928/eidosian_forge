from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.vertexai import get_client_info
def _prepare_request_metadata(self, user_ldap: str) -> 'RequestMetadata':
    from google.cloud.contentwarehouse_v1 import RequestMetadata, UserInfo
    user_info = UserInfo(id=f'user:{user_ldap}')
    return RequestMetadata(user_info=user_info)