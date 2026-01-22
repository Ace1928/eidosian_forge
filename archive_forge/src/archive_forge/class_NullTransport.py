class NullTransport:

    async def connect_tcp(self, host, port, timeout, local_address):
        raise NotImplementedError