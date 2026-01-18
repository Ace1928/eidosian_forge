from typing import Dict
import requests
from langchain_core.pydantic_v1 import BaseModel, BaseSettings, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
Run SceneXplain image explainer.