from __future__ import annotations
from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict
class IntegrationWandb(TypedDict, total=False):
    project: Required[str]
    'The name of the project that the new run will be created under.'
    entity: Optional[str]
    'The entity to use for the run.\n\n    This allows you to set the team or username of the WandB user that you would\n    like associated with the run. If not set, the default entity for the registered\n    WandB API key is used.\n    '
    name: Optional[str]
    'A display name to set for the run.\n\n    If not set, we will use the Job ID as the name.\n    '
    tags: List[str]
    'A list of tags to be attached to the newly created run.\n\n    These tags are passed through directly to WandB. Some default tags are generated\n    by OpenAI: "openai/finetune", "openai/{base-model}", "openai/{ftjob-abcdef}".\n    '