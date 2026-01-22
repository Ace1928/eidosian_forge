import dns._features
import dns.asyncbackend
def _asyncio_manager_factory(context, *args, **kwargs):
    return AsyncioQuicManager(*args, **kwargs)