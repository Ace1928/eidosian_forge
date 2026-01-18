from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_supports_loading_multiple_keys_in_one_call():

    def call_fn(keys):
        return Promise.resolve(keys)
    identity_loader = DataLoader(call_fn)
    promise_all = identity_loader.load_many([1, 2])
    assert isinstance(promise_all, Promise)
    values = promise_all.get()
    assert values == [1, 2]
    promise_all = identity_loader.load_many([])
    assert isinstance(promise_all, Promise)
    values = promise_all.get()
    assert values == []