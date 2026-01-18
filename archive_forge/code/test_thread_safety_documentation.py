from promise import Promise
from promise.dataloader import DataLoader
import threading

    Dataloader should only batch `load` calls that happened on the same thread.
    
    Here we assert that `load` calls on thread 2 are not batched on thread 1 as
    thread 1 batches its own `load` calls.
    