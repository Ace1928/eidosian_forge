from __future__ import annotations as _annotations
import warnings
from contextlib import contextmanager
from typing import (
from pydantic_core import core_schema
from typing_extensions import (
from ..aliases import AliasGenerator
from ..config import ConfigDict, ExtraValues, JsonDict, JsonEncoder, JsonSchemaExtraCallable
from ..errors import PydanticUserError
from ..warnings import PydanticDeprecatedSince20
class ConfigWrapper:
    """Internal wrapper for Config which exposes ConfigDict items as attributes."""
    __slots__ = ('config_dict',)
    config_dict: ConfigDict
    title: str | None
    str_to_lower: bool
    str_to_upper: bool
    str_strip_whitespace: bool
    str_min_length: int
    str_max_length: int | None
    extra: ExtraValues | None
    frozen: bool
    populate_by_name: bool
    use_enum_values: bool
    validate_assignment: bool
    arbitrary_types_allowed: bool
    from_attributes: bool
    loc_by_alias: bool
    alias_generator: Callable[[str], str] | AliasGenerator | None
    ignored_types: tuple[type, ...]
    allow_inf_nan: bool
    json_schema_extra: JsonDict | JsonSchemaExtraCallable | None
    json_encoders: dict[type[object], JsonEncoder] | None
    strict: bool
    revalidate_instances: Literal['always', 'never', 'subclass-instances']
    ser_json_timedelta: Literal['iso8601', 'float']
    ser_json_bytes: Literal['utf8', 'base64']
    ser_json_inf_nan: Literal['null', 'constants']
    validate_default: bool
    validate_return: bool
    protected_namespaces: tuple[str, ...]
    hide_input_in_errors: bool
    defer_build: bool
    plugin_settings: dict[str, object] | None
    schema_generator: type[GenerateSchema] | None
    json_schema_serialization_defaults_required: bool
    json_schema_mode_override: Literal['validation', 'serialization', None]
    coerce_numbers_to_str: bool
    regex_engine: Literal['rust-regex', 'python-re']
    validation_error_cause: bool

    def __init__(self, config: ConfigDict | dict[str, Any] | type[Any] | None, *, check: bool=True):
        if check:
            self.config_dict = prepare_config(config)
        else:
            self.config_dict = cast(ConfigDict, config)

    @classmethod
    def for_model(cls, bases: tuple[type[Any], ...], namespace: dict[str, Any], kwargs: dict[str, Any]) -> Self:
        """Build a new `ConfigWrapper` instance for a `BaseModel`.

        The config wrapper built based on (in descending order of priority):
        - options from `kwargs`
        - options from the `namespace`
        - options from the base classes (`bases`)

        Args:
            bases: A tuple of base classes.
            namespace: The namespace of the class being created.
            kwargs: The kwargs passed to the class being created.

        Returns:
            A `ConfigWrapper` instance for `BaseModel`.
        """
        config_new = ConfigDict()
        for base in bases:
            config = getattr(base, 'model_config', None)
            if config:
                config_new.update(config.copy())
        config_class_from_namespace = namespace.get('Config')
        config_dict_from_namespace = namespace.get('model_config')
        if config_class_from_namespace and config_dict_from_namespace:
            raise PydanticUserError('"Config" and "model_config" cannot be used together', code='config-both')
        config_from_namespace = config_dict_from_namespace or prepare_config(config_class_from_namespace)
        config_new.update(config_from_namespace)
        for k in list(kwargs.keys()):
            if k in config_keys:
                config_new[k] = kwargs.pop(k)
        return cls(config_new)
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            try:
                return self.config_dict[name]
            except KeyError:
                try:
                    return config_defaults[name]
                except KeyError:
                    raise AttributeError(f'Config has no attribute {name!r}') from None

    def core_config(self, obj: Any) -> core_schema.CoreConfig:
        """Create a pydantic-core config, `obj` is just used to populate `title` if not set in config.

        Pass `obj=None` if you do not want to attempt to infer the `title`.

        We don't use getattr here since we don't want to populate with defaults.

        Args:
            obj: An object used to populate `title` if not set in config.

        Returns:
            A `CoreConfig` object created from config.
        """

        def dict_not_none(**kwargs: Any) -> Any:
            return {k: v for k, v in kwargs.items() if v is not None}
        core_config = core_schema.CoreConfig(**dict_not_none(title=self.config_dict.get('title') or (obj and obj.__name__), extra_fields_behavior=self.config_dict.get('extra'), allow_inf_nan=self.config_dict.get('allow_inf_nan'), populate_by_name=self.config_dict.get('populate_by_name'), str_strip_whitespace=self.config_dict.get('str_strip_whitespace'), str_to_lower=self.config_dict.get('str_to_lower'), str_to_upper=self.config_dict.get('str_to_upper'), strict=self.config_dict.get('strict'), ser_json_timedelta=self.config_dict.get('ser_json_timedelta'), ser_json_bytes=self.config_dict.get('ser_json_bytes'), ser_json_inf_nan=self.config_dict.get('ser_json_inf_nan'), from_attributes=self.config_dict.get('from_attributes'), loc_by_alias=self.config_dict.get('loc_by_alias'), revalidate_instances=self.config_dict.get('revalidate_instances'), validate_default=self.config_dict.get('validate_default'), str_max_length=self.config_dict.get('str_max_length'), str_min_length=self.config_dict.get('str_min_length'), hide_input_in_errors=self.config_dict.get('hide_input_in_errors'), coerce_numbers_to_str=self.config_dict.get('coerce_numbers_to_str'), regex_engine=self.config_dict.get('regex_engine'), validation_error_cause=self.config_dict.get('validation_error_cause')))
        return core_config

    def __repr__(self):
        c = ', '.join((f'{k}={v!r}' for k, v in self.config_dict.items()))
        return f'ConfigWrapper({c})'