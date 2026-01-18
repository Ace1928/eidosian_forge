import base64
import inspect
import json
import re
import urllib
from copy import deepcopy
from typing import List as LList
from .... import __version__ as wandb_ver
from .... import termlog, termwarn
from ....sdk.lib import ipython
from ...public import Api as PublicApi
from ...public import RetryingClient
from ._blocks import P, PanelGrid, UnknownBlock, WeaveBlock, block_mapping, weave_blocks
from .mutations import UPSERT_VIEW, VIEW_REPORT
from .runset import Runset
from .util import Attr, Base, Block, coalesce, generate_name, nested_get, nested_set
from .validators import OneOf, TypeValidator

                Server does not generate IDs with correct padding, so decode with default validate=False.
                Then re-encode it with correct padding.
                https://stackoverflow.com/questions/2941995/python-ignore-incorrect-padding-error-when-base64-decoding

                Corresponding core app logic that strips the padding in url
                https://github.com/wandb/core/blob/b563437c1f3237ec35b1fb388ac14abbab7b4279/frontends/app/src/util/url/shared.ts#L33-L78
            