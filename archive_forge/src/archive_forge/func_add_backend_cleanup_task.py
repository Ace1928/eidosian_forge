from celery._state import connect_on_app_finalize
from celery.utils.log import get_logger
@connect_on_app_finalize
def add_backend_cleanup_task(app):
    """Task used to clean up expired results.

    If the configured backend requires periodic cleanup this task is also
    automatically configured to run every day at 4am (requires
    :program:`celery beat` to be running).
    """

    @app.task(name='celery.backend_cleanup', shared=False, lazy=False)
    def backend_cleanup():
        app.backend.cleanup()
    return backend_cleanup