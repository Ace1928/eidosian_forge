import logging
import threading
import time
import numpy as np
Changes the steps per execution using the following algorithm.

        If there is more than a 10% increase in the throughput, then the last
        recorded action is repeated (i.e. if increasing the SPE caused an
        increase in throughput, it is increased again). If there is more than a
        10% decrease in the throughput, then the opposite of the last action is
        performed (i.e. if increasing the SPE decreased the throughput, then the
        SPE is decreased).
        