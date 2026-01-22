from celery._state import connect_on_app_finalize
from celery.utils.log import get_logger
@connect_on_app_finalize
def add_group_task(app):
    """No longer used, but here for backwards compatibility."""
    from celery.canvas import maybe_signature
    from celery.result import result_from_tuple

    @app.task(name='celery.group', bind=True, shared=False, lazy=False)
    def group(self, tasks, result, group_id, partial_args, add_to_parent=True):
        app = self.app
        result = result_from_tuple(result, app)
        taskit = (maybe_signature(task, app=app).clone(partial_args) for i, task in enumerate(tasks))
        with app.producer_or_acquire() as producer:
            [stask.apply_async(group_id=group_id, producer=producer, add_to_parent=False) for stask in taskit]
        parent = app.current_worker_task
        if add_to_parent and parent:
            parent.add_trail(result)
        return result
    return group