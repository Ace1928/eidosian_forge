class TaskStubRepo(object):

    def __init__(self, base, task_stub_proxy_class=None, task_stub_proxy_kwargs=None):
        self.base = base
        self.task_stub_proxy_helper = Helper(task_stub_proxy_class, task_stub_proxy_kwargs)

    def list(self, *args, **kwargs):
        tasks = self.base.list(*args, **kwargs)
        return [self.task_stub_proxy_helper.proxy(task) for task in tasks]