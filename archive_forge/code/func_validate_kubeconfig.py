import pathlib
from typing import Optional, Union
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
@validator('kubeconfig', pre=True)
def validate_kubeconfig(cls, v):
    if v is not None:
        return v
    from lazyops.utils.system import get_k8s_kubeconfig
    return get_k8s_kubeconfig()