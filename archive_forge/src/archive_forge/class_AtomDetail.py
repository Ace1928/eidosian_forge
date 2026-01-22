import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
class AtomDetail(object, metaclass=abc.ABCMeta):
    """A collection of atom specific runtime information and metadata.

    This is a base **abstract** class that contains attributes that are used
    to connect a atom to the persistence layer before, during, or after it is
    running. It includes any results it may have produced, any state that it
    may be in (for example ``FAILURE``), any exception that occurred when
    running, and any associated stacktrace that may have occurring during an
    exception being thrown. It may also contain any other metadata that
    should also be stored along-side the details about the connected atom.

    The data contained within this class need **not** be persisted to the
    backend storage in real time. The data in this class will only be
    guaranteed to be persisted when a save (or update) occurs via some backend
    connection.

    :ivar intention: The execution strategy of the atom associated
                     with this atom detail (used by an engine/others to
                     determine if the associated atom needs to be
                     executed, reverted, retried and so-on).
    :ivar meta: A dictionary of meta-data associated with this atom detail.
    :ivar version: A version tuple or string that represents the
                   atom version this atom detail is associated with (typically
                   used for introspection and any data migration
                   strategies).
    :ivar results: Any results the atom produced from either its
                   ``execute`` method or from other sources.
    :ivar revert_results: Any results the atom produced from either its
                          ``revert`` method or from other sources.
    :ivar AtomDetail.failure: If the atom failed (due to its ``execute`` method
                   raising) this will be a
                   :py:class:`~taskflow.types.failure.Failure` object that
                   represents that failure (if there was no failure this
                   will be set to none).
    :ivar revert_failure: If the atom failed (possibly due to its ``revert``
                          method raising) this will be a
                          :py:class:`~taskflow.types.failure.Failure` object
                          that represents that failure (if there was no
                          failure this will be set to none).
    """

    def __init__(self, name, uuid):
        self._uuid = uuid
        self._name = name
        self.state = None
        self.intention = states.EXECUTE
        self.results = None
        self.failure = None
        self.revert_results = None
        self.revert_failure = None
        self.meta = {}
        self.version = None

    @property
    def last_results(self):
        """Gets the atoms last result.

        If the atom has produced many results (for example if it has been
        retried, reverted, executed and ...) this returns the last one of
        many results.
        """
        return self.results

    def update(self, ad):
        """Updates the object's state to be the same as the given one.

        This will assign the private and public attributes of the given
        atom detail directly to this object (replacing any existing
        attributes in this object; even if they are the **same**).

        NOTE(harlowja): If the provided object is this object itself
        then **no** update is done.

        :returns: this atom detail
        :rtype: :py:class:`.AtomDetail`
        """
        if ad is self:
            return self
        self.state = ad.state
        self.intention = ad.intention
        self.meta = ad.meta
        self.failure = ad.failure
        self.results = ad.results
        self.revert_results = ad.revert_results
        self.revert_failure = ad.revert_failure
        self.version = ad.version
        return self

    @abc.abstractmethod
    def merge(self, other, deep_copy=False):
        """Merges the current object state with the given ones state.

        If ``deep_copy`` is provided as truthy then the
        local object will use ``copy.deepcopy`` to replace this objects
        local attributes with the provided objects attributes (**only** if
        there is a difference between this objects attributes and the
        provided attributes). If ``deep_copy`` is falsey (the default) then a
        reference copy will occur instead when a difference is detected.

        NOTE(harlowja): If the provided object is this object itself
        then **no** merging is done. Do note that **no** results are merged
        in this method. That operation **must** to be the responsibilty of
        subclasses to implement and override this abstract method
        and provide that merging themselves as they see fit.

        :returns: this atom detail (freshly merged with the incoming object)
        :rtype: :py:class:`.AtomDetail`
        """
        copy_fn = _copy_function(deep_copy)
        self.state = other.state
        self.intention = other.intention
        if self.failure != other.failure:
            if other.failure:
                if deep_copy:
                    self.failure = other.failure.copy()
                else:
                    self.failure = other.failure
            else:
                self.failure = None
        if self.revert_failure != other.revert_failure:
            if other.revert_failure:
                if deep_copy:
                    self.revert_failure = other.revert_failure.copy()
                else:
                    self.revert_failure = other.revert_failure
            else:
                self.revert_failure = None
        if self.meta != other.meta:
            self.meta = copy_fn(other.meta)
        if self.version != other.version:
            self.version = copy_fn(other.version)
        return self

    @abc.abstractmethod
    def put(self, state, result):
        """Puts a result (acquired in the given state) into this detail."""

    def to_dict(self):
        """Translates the internal state of this object to a ``dict``.

        :returns: this atom detail in ``dict`` form
        """
        if self.failure:
            failure = self.failure.to_dict()
        else:
            failure = None
        if self.revert_failure:
            revert_failure = self.revert_failure.to_dict()
        else:
            revert_failure = None
        return {'failure': failure, 'revert_failure': revert_failure, 'meta': self.meta, 'name': self.name, 'results': self.results, 'revert_results': self.revert_results, 'state': self.state, 'version': self.version, 'intention': self.intention, 'uuid': self.uuid}

    @classmethod
    def from_dict(cls, data):
        """Translates the given ``dict`` into an instance of this class.

        NOTE(harlowja): the ``dict`` provided should come from a prior
        call to :meth:`.to_dict`.

        :returns: a new atom detail
        :rtype: :py:class:`.AtomDetail`
        """
        obj = cls(data['name'], data['uuid'])
        obj.state = data.get('state')
        obj.intention = data.get('intention')
        obj.results = data.get('results')
        obj.revert_results = data.get('revert_results')
        obj.version = data.get('version')
        obj.meta = _fix_meta(data)
        failure = data.get('failure')
        if failure:
            obj.failure = ft.Failure.from_dict(failure)
        revert_failure = data.get('revert_failure')
        if revert_failure:
            obj.revert_failure = ft.Failure.from_dict(revert_failure)
        return obj

    @property
    def uuid(self):
        """The unique identifer of this atom detail."""
        return self._uuid

    @property
    def name(self):
        """The name of this atom detail."""
        return self._name

    @abc.abstractmethod
    def reset(self, state):
        """Resets this atom detail and sets ``state`` attribute value."""

    @abc.abstractmethod
    def copy(self):
        """Copies this atom detail."""

    def pformat(self, indent=0, linesep=os.linesep):
        """Pretty formats this atom detail into a string."""
        cls_name = self.__class__.__name__
        lines = ["%s%s: '%s'" % (' ' * indent, cls_name, self.name)]
        lines.extend(_format_shared(self, indent=indent + 1))
        lines.append('%s- version = %s' % (' ' * (indent + 1), misc.get_version_string(self)))
        lines.append('%s- results = %s' % (' ' * (indent + 1), self.results))
        lines.append('%s- failure = %s' % (' ' * (indent + 1), bool(self.failure)))
        lines.extend(_format_meta(self.meta, indent=indent + 1))
        return linesep.join(lines)