from ._base import *
import operator as op
class LazyUserSchema(BaseModel):
    role: str = ...
    is_super: bool = False
    username: str = ...
    password: str = ...
    hash_password: str = None

    @classmethod
    def get_schema(cls, config: Dict[str, Any]=None, is_dev: bool=True, *args, **kwargs):
        current_schema = LazyUserSchema.__fields__
        schema_data = {field: (vals.type_, ...) if vals.required else (vals.type_, vals.default) for field, vals in current_schema.items()}
        if not is_dev:
            schema_data.pop('password')
        if not config:
            return schema_data
        new_schema = config if isinstance(config, dict) else config.dict(exclude_unset=True)
        for field, values in new_schema.items():
            schema_data[field] = values
        return schema_data

    @classmethod
    def get_hash_schema(cls):
        return {'password': 'hash_password'}