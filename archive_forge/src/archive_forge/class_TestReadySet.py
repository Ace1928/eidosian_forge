import duet
import duet.impl as impl
class TestReadySet:

    def test_get_all_returns_all_ready_tasks(self):
        task1 = make_task(duet.completed_future(None))
        task2 = make_task(duet.completed_future(None))
        task3 = make_task(duet.AwaitableFuture())
        task4 = make_task(duet.completed_future(None))
        rs = impl.ReadySet()
        rs.register(task1)
        rs.register(task2)
        rs.register(task3)
        rs.register(task4)
        tasks = rs.get_all()
        assert tasks == [task1, task2, task4]

    def test_task_added_at_most_once(self):
        task = make_task(duet.completed_future(None))
        rs = impl.ReadySet()
        rs.register(task)
        rs.register(task)
        tasks = rs.get_all()
        assert tasks == [task]

    def test_futures_flushed_if_no_task_ready(self):
        future = CompleteOnFlush()
        task = make_task(future)
        rs = impl.ReadySet()
        rs.register(task)
        tasks = rs.get_all()
        assert tasks == [task]
        assert future.flushed

    def test_futures_not_flushed_if_tasks_ready(self):
        future = CompleteOnFlush()
        task1 = make_task(future)
        task2 = make_task(duet.completed_future(None))
        rs = impl.ReadySet()
        rs.register(task1)
        rs.register(task2)
        tasks = rs.get_all()
        assert tasks == [task2]
        assert not future.flushed