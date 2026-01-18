import duet
import duet.impl as impl
def test_task_added_at_most_once(self):
    task = make_task(duet.completed_future(None))
    rs = impl.ReadySet()
    rs.register(task)
    rs.register(task)
    tasks = rs.get_all()
    assert tasks == [task]