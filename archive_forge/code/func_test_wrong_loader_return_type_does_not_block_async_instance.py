from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_wrong_loader_return_type_does_not_block_async_instance():

    @Promise.safe
    def do():

        def do_resolve(x):
            return x
        a_loader, a_load_calls = id_loader(resolve=do_resolve)
        with raises(Exception):
            a_loader.load('A1').get()
        assert async_instance.have_drained_queues
        with raises(Exception):
            a_loader.load('A2').get()
        assert async_instance.have_drained_queues
    do().get()