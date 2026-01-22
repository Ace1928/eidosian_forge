from celery._state import connect_on_app_finalize
from celery.utils.log import get_logger
@connect_on_app_finalize
def add_chain_task(app):
    """No longer used, but here for backwards compatibility."""

    @app.task(name='celery.chain', shared=False, lazy=False)
    def chain(*args, **kwargs):
        raise NotImplementedError('chain is not a real task')
    return chain