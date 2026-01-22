import dns._features
import dns.asyncbackend
class AsyncQuicConnection:

    async def make_stream(self) -> Any:
        raise NotImplementedError