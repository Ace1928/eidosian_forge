from promise import Promise
from promise.dataloader import DataLoader
import threading
def then_1(value):
    promise = Promise.resolve(None).then(then_2)
    assert promise.is_pending
    event_1.set()
    event_2.wait()