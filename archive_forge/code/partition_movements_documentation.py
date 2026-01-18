import logging
from collections import defaultdict, namedtuple
from copy import deepcopy

    This class maintains some data structures to simplify lookup of partition movements
    among consumers. At each point of time during a partition rebalance it keeps track
    of partition movements corresponding to each topic, and also possible movement (in
    form a ConsumerPair object) for each partition.
    