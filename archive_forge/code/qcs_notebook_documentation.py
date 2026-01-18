import dataclasses
from typing import cast, Optional, Sequence, Union
import cirq
from cirq_google import ProcessorSampler, get_engine
from cirq_google.engine import (
Authenticates on Google Cloud and returns Engine related objects.

    This function will authenticate to Google Cloud and attempt to
    instantiate an Engine object.  If it does not succeed, it will instead
    return a virtual AbstractEngine that is backed by a noisy simulator.
    This function is designed for maximum versatility and
    to work in colab notebooks, as a stand-alone, and in tests.

    Note that, if you are using this to connect to QCS and do not care about
    the added versatility, you may want to use `cirq_google.get_engine()` or
    `cirq_google.Engine()` instead to guarantee the use of a production instance
    and to avoid accidental use of a noisy simulator.

    Args:
        project_id: Optional explicit Google Cloud project id. Otherwise,
            this defaults to the environment variable GOOGLE_CLOUD_PROJECT.
            By using an environment variable, you can avoid hard-coding
            personal project IDs in shared code.
        processor_id: Engine processor ID (from Cloud console or
            ``Engine.list_processors``).
        virtual: If set to True, will create a noisy virtual Engine instead.
            This is useful for testing and simulation.

    Returns:
        An instance of QCSObjectsForNotebook which contains all the objects .

    Raises:
        ValueError: if processor_id is not specified and no processors are available.
    