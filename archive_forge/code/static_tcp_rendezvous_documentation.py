import datetime
import logging
from typing import Tuple, cast, Optional
from torch.distributed import Store, TCPStore, PrefixStore
from torch.distributed.elastic.rendezvous import RendezvousHandler, RendezvousParameters
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint

    Static rendezvous that is a wrapper around the TCPStore.

    Creates TCPStore based on the input parameters with the
    listener on the agent with group_rank=0
    