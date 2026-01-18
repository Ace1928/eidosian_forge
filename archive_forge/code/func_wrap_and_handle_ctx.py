import logging
import os
from os_ken.lib import ip
def wrap_and_handle_ctx(sock, addr):
    handle(ctx.wrap_socket(sock, **ssl_args), addr)