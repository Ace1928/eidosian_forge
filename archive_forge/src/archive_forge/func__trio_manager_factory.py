import dns._features
import dns.asyncbackend
def _trio_manager_factory(context, *args, **kwargs):
    return TrioQuicManager(context, *args, **kwargs)