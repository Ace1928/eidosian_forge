import time
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict
def to_api_key_claims(self) -> 'APIKeyJWTClaims':
    """
        Converts the User JWT Claims to an API Key JWT Claims
        """
    return APIKeyJWTClaims.model_validate(self, from_attributes=True)