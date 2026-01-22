from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union
from wasabi import msg
class ProjectConfigSchema(BaseModel):
    vars: Dict[StrictStr, Any] = Field({}, title='Optional variables to substitute in commands')
    env: Dict[StrictStr, Any] = Field({}, title='Optional variable names to substitute in commands, mapped to environment variable names')
    assets: List[Union[ProjectConfigAssetURL, ProjectConfigAssetGit]] = Field([], title='Data assets')
    workflows: Dict[StrictStr, List[StrictStr]] = Field({}, title='Named workflows, mapped to list of project commands to run in order')
    commands: List[ProjectConfigCommand] = Field([], title='Project command shortucts')
    title: Optional[str] = Field(None, title='Project title')

    class Config:
        title = 'Schema for project configuration file'

    @root_validator(pre=True)
    def check_legacy_keys(cls, obj: Dict[str, Any]) -> Dict[str, Any]:
        if 'spacy_version' in obj:
            msg.warn('Your project configuration file includes a `spacy_version` key, which is now deprecated. Weasel will not validate your version of spaCy.')
        if 'check_requirements' in obj:
            msg.warn('Your project configuration file includes a `check_requirements` key, which is now deprecated. Weasel will not validate your requirements.')
        return obj