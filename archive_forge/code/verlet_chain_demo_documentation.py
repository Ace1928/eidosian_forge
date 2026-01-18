import numpy as np
import verlet_chain
import pyqtgraph as pg

Mechanical simulation of a chain using verlet integration.

Use the mouse to interact with one of the chains.

By default, this uses a slow, pure-python integrator to solve the chain link
positions. Unix users may compile a small math library to speed this up by 
running the `examples/verlet_chain/make` script.

