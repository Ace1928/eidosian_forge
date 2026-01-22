import threading
from typing import TYPE_CHECKING, Any, Generator, Iterator, List, Optional, Tuple
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.store import Store


This wrapper intercepts calls through the store interface and implements
thread-safe logging of destructive operations (adds / removes) in reverse.
This is persisted on the store instance and the reverse operations are
executed In order to return the store to the state it was when the transaction
began Since the reverse operations are persisted on the store, the store
itself acts as a transaction.

Calls to commit or rollback, flush the list of reverse operations This
provides thread-safe atomicity and isolation (assuming concurrent operations
occur with different store instances), but no durability (transactions are
persisted in memory and wont  be available to reverse operations after the
system fails): A and I out of ACID.

