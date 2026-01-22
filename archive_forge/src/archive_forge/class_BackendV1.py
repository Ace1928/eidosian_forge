from abc import ABC
from abc import abstractmethod
import datetime
from typing import List, Union, Iterable, Tuple
from qiskit.providers.provider import Provider
from qiskit.providers.models.backendstatus import BackendStatus
from qiskit.circuit.gate import Instruction
class BackendV1(Backend, ABC):
    """Abstract class for Backends

    This abstract class is to be used for all Backend objects created by a
    provider. There are several classes of information contained in a Backend.
    The first are the attributes of the class itself. These should be used to
    defined the immutable characteristics of the backend. The ``options``
    attribute of the backend is used to contain the dynamic user configurable
    options of the backend. It should be used more for runtime options
    that configure how the backend is used. For example, something like a
    ``shots`` field for a backend that runs experiments which would contain an
    int for how many shots to execute. The ``properties`` attribute is
    optionally defined :class:`~qiskit.providers.models.BackendProperties`
    object and is used to return measured properties, or properties
    of a backend that may change over time. The simplest example of this would
    be a version string, which will change as a backend is updated, but also
    could be something like noise parameters for backends that run experiments.

    This first version of the Backend abstract class is written to be mostly
    backwards compatible with the legacy providers interface. This includes reusing
    the model objects :class:`~qiskit.providers.models.BackendProperties` and
    :class:`~qiskit.providers.models.BackendConfiguration`. This was done to
    ease the transition for users and provider maintainers to the new versioned providers.
    Expect, future versions of this abstract class to change the data model and
    interface.

    Subclasses of this should override the public method :meth:`run` and the internal
    :meth:`_default_options`:

    .. automethod:: _default_options
    """
    version = 1

    def __init__(self, configuration, provider=None, **fields):
        """Initialize a backend class

        Args:
            configuration (BackendConfiguration): A backend configuration
                object for the backend object.
            provider (qiskit.providers.Provider): Optionally, the provider
                object that this Backend comes from.
            fields: kwargs for the values to use to override the default
                options.
        Raises:
            AttributeError: if input field not a valid options

        ..
            This next bit is necessary just because autosummary generally won't summarise private
            methods; changing that behaviour would have annoying knock-on effects through all the
            rest of the documentation, so instead we just hard-code the automethod directive.

        In addition to the public abstract methods, subclasses should also implement the following
        private methods:

        .. automethod:: _default_options
           :noindex:
        """
        self._configuration = configuration
        self._options = self._default_options()
        self._provider = provider
        if fields:
            for field in fields:
                if field not in self._options.data:
                    raise AttributeError('Options field %s is not valid for this backend' % field)
            self._options.update_config(**fields)

    @classmethod
    @abstractmethod
    def _default_options(cls):
        """Return the default options

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.Options: A options object with
                default values set
        """

    def set_options(self, **fields):
        """Set the options fields for the backend

        This method is used to update the options of a backend. If
        you need to change any of the options prior to running just
        pass in the kwarg with the new value for the options.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If the field passed in is not part of the
                options
        """
        for field in fields:
            if not hasattr(self._options, field):
                raise AttributeError('Options field %s is not valid for this backend' % field)
        self._options.update_options(**fields)

    def configuration(self):
        """Return the backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        return self._configuration

    def properties(self):
        """Return the backend properties.

        Returns:
            BackendProperties: the configuration for the backend. If the backend
            does not support properties, it returns ``None``.
        """
        return None

    def provider(self):
        """Return the backend Provider.

        Returns:
            Provider: the Provider responsible for the backend.
        """
        return self._provider

    def status(self):
        """Return the backend status.

        Returns:
            BackendStatus: the status of the backend.
        """
        return BackendStatus(backend_name=self.name(), backend_version='1', operational=True, pending_jobs=0, status_msg='')

    def name(self):
        """Return the backend name.

        Returns:
            str: the name of the backend.
        """
        return self._configuration.backend_name

    def __str__(self):
        return self.name()

    def __repr__(self):
        """Official string representation of a Backend.

        Note that, by Qiskit convention, it is consciously *not* a fully valid
        Python expression. Subclasses should provide 'a string of the form
        <...some useful description...>'. [0]

        [0] https://docs.python.org/3/reference/datamodel.html#object.__repr__
        """
        return f"<{self.__class__.__name__}('{self.name()}')>"

    @property
    def options(self):
        """Return the options for the backend

        The options of a backend are the dynamic parameters defining
        how the backend is used. These are used to control the :meth:`run`
        method.
        """
        return self._options

    @abstractmethod
    def run(self, run_input, **options):
        """Run on the backend.

        This method returns a :class:`~qiskit.providers.Job` object
        that runs circuits. Depending on the backend this may be either an async
        or sync call. It is at the discretion of the provider to decide whether
        running should block until the execution is finished or not: the Job
        class can handle either situation.

        Args:
            run_input (QuantumCircuit or Schedule or list): An individual or a
                list of :class:`~qiskit.circuit.QuantumCircuit` or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
                For legacy providers migrating to the new versioned providers,
                provider interface a :class:`~qiskit.qobj.QasmQobj` or
                :class:`~qiskit.qobj.PulseQobj` objects should probably be
                supported too (but deprecated) for backwards compatibility. Be
                sure to update the docstrings of subclasses implementing this
                method to document that. New provider implementations should not
                do this though as :mod:`qiskit.qobj` will be deprecated and
                removed along with the legacy providers interface.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.
        Returns:
            Job: The job object for the run
        """
        pass