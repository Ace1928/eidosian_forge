from dataclasses import InitVar
from typing import Type, Any, Optional, Union, Collection, TypeVar, Dict, Callable, Mapping, List, Tuple
def transform_value(type_hooks: Dict[Type, Callable[[Any], Any]], cast: List[Type], target_type: Type, value: Any) -> Any:
    if target_type in type_hooks:
        value = type_hooks[target_type](value)
    else:
        for cast_type in cast:
            if is_subclass(target_type, cast_type):
                if is_generic_collection(target_type):
                    value = extract_origin_collection(target_type)(value)
                else:
                    value = target_type(value)
                break
    if is_optional(target_type):
        if value is None:
            return None
        target_type = extract_optional(target_type)
        return transform_value(type_hooks, cast, target_type, value)
    if is_generic_collection(target_type) and isinstance(value, extract_origin_collection(target_type)):
        collection_cls = value.__class__
        if issubclass(collection_cls, dict):
            key_cls, item_cls = extract_generic(target_type, defaults=(Any, Any))
            return collection_cls({transform_value(type_hooks, cast, key_cls, key): transform_value(type_hooks, cast, item_cls, item) for key, item in value.items()})
        item_cls = extract_generic(target_type, defaults=(Any,))[0]
        return collection_cls((transform_value(type_hooks, cast, item_cls, item) for item in value))
    return value