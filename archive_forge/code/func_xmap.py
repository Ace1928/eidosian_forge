from celery._state import connect_on_app_finalize
from celery.utils.log import get_logger
@app.task(name='celery.map', shared=False, lazy=False)
def xmap(task, it):
    task = signature(task, app=app).type
    return [task(item) for item in it]