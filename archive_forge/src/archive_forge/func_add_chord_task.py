from celery._state import connect_on_app_finalize
from celery.utils.log import get_logger
@connect_on_app_finalize
def add_chord_task(app):
    """No longer used, but here for backwards compatibility."""
    from celery import chord as _chord
    from celery import group
    from celery.canvas import maybe_signature

    @app.task(name='celery.chord', bind=True, ignore_result=False, shared=False, lazy=False)
    def chord(self, header, body, partial_args=(), interval=None, countdown=1, max_retries=None, eager=False, **kwargs):
        app = self.app
        tasks = header.tasks if isinstance(header, group) else header
        header = group([maybe_signature(s, app=app) for s in tasks], app=self.app)
        body = maybe_signature(body, app=app)
        ch = _chord(header, body)
        return ch.run(header, body, partial_args, app, interval, countdown, max_retries, **kwargs)
    return chord