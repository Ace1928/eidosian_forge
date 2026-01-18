import inspect
import re
import warnings
from functools import wraps
from itertools import chain
from typing import Any, Dict
from ._typing import CallableT
def validate_hf_hub_args(fn: CallableT) -> CallableT:
    """Validate values received as argument for any public method of `huggingface_hub`.

    The goal of this decorator is to harmonize validation of arguments reused
    everywhere. By default, all defined validators are tested.

    Validators:
        - [`~utils.validate_repo_id`]: `repo_id` must be `"repo_name"`
          or `"namespace/repo_name"`. Namespace is a username or an organization.
        - [`~utils.smoothly_deprecate_use_auth_token`]: Use `token` instead of
          `use_auth_token` (only if `use_auth_token` is not expected by the decorated
          function - in practice, always the case in `huggingface_hub`).

    Example:
    ```py
    >>> from huggingface_hub.utils import validate_hf_hub_args

    >>> @validate_hf_hub_args
    ... def my_cool_method(repo_id: str):
    ...     print(repo_id)

    >>> my_cool_method(repo_id="valid_repo_id")
    valid_repo_id

    >>> my_cool_method("other..repo..id")
    huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

    >>> my_cool_method(repo_id="other..repo..id")
    huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

    >>> @validate_hf_hub_args
    ... def my_cool_auth_method(token: str):
    ...     print(token)

    >>> my_cool_auth_method(token="a token")
    "a token"

    >>> my_cool_auth_method(use_auth_token="a use_auth_token")
    "a use_auth_token"

    >>> my_cool_auth_method(token="a token", use_auth_token="a use_auth_token")
    UserWarning: Both `token` and `use_auth_token` are passed (...)
    "a token"
    ```

    Raises:
        [`~utils.HFValidationError`]:
            If an input is not valid.
    """
    signature = inspect.signature(fn)
    check_use_auth_token = 'use_auth_token' not in signature.parameters and 'token' in signature.parameters

    @wraps(fn)
    def _inner_fn(*args, **kwargs):
        has_token = False
        for arg_name, arg_value in chain(zip(signature.parameters, args), kwargs.items()):
            if arg_name in ['repo_id', 'from_id', 'to_id']:
                validate_repo_id(arg_value)
            elif arg_name == 'token' and arg_value is not None:
                has_token = True
        if check_use_auth_token:
            kwargs = smoothly_deprecate_use_auth_token(fn_name=fn.__name__, has_token=has_token, kwargs=kwargs)
        return fn(*args, **kwargs)
    return _inner_fn