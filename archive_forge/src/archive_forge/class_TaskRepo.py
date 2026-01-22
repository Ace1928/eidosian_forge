class TaskRepo(object):

    def __init__(self, base, task_proxy_class=None, task_proxy_kwargs=None):
        self.base = base
        self.task_proxy_helper = Helper(task_proxy_class, task_proxy_kwargs)

    def get(self, task_id):
        task = self.base.get(task_id)
        return self.task_proxy_helper.proxy(task)

    def add(self, task):
        self.base.add(self.task_proxy_helper.unproxy(task))

    def save(self, task):
        self.base.save(self.task_proxy_helper.unproxy(task))

    def remove(self, task):
        base_task = self.task_proxy_helper.unproxy(task)
        self.base.remove(base_task)