import asyncio
import logging
import traceback
from typing import Union
from rpcq._utils import rpc_reply, rpc_error, RPCMethodError, get_input, get_safe_input, \
from rpcq.messages import RPCRequest, RPCReply, RPCError

        Process a JSON RPC request

        :param RPCRequest request: JSON RPC request
        :return: JSON RPC reply
        