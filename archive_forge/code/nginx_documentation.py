from ._base import *

        Write an nginx configuration to a file-like object.
        :param obj obj: nginx object (NginxConfig, Server, Container)
        :param obj fobj: file-like object to write to
        :returns: file-like object that was written to
        