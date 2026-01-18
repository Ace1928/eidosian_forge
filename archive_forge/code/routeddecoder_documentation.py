from io import BufferedIOBase
from typing import Any, Callable, Iterable, Iterator, Sized, Tuple
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import _deprecation_warning
from torch.utils.data.datapipes.utils.decoder import (

    Decodes binary streams from input DataPipe, yields pathname and decoded data in a tuple.

    (functional name: ``routed_decode``)

    Args:
        datapipe: Iterable datapipe that provides pathname and binary stream in tuples
        handlers: Optional user defined decoder handlers. If ``None``, basic and image decoder
            handlers will be set as default. If multiple handles are provided, the priority
            order follows the order of handlers (the first handler has the top priority)
        key_fn: Function for decoder to extract key from pathname to dispatch handlers.
            Default is set to extract file extension from pathname

    Note:
        When ``key_fn`` is specified returning anything other than extension, the default
        handler will not work and users need to specify custom handler. Custom handler
        could use regex to determine the eligibility to handle data.
    