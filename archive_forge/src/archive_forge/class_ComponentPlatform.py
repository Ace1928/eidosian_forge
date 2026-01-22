from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
class ComponentPlatform(object):
    """Information on the applicable platforms for the component.

  Attributes:
    operating_systems: [platforms.OperatingSystem], The operating systems this
      component is valid on.  If [] or None, it is valid on all operating
      systems.
    architectures: [platforms.Architecture], The architectures this component is
      valid on.  If [] or None, it is valid on all architectures.
  """

    @classmethod
    def FromDictionary(cls, dictionary):
        """Parses operating_systems and architectures from a dictionary."""
        p = DictionaryParser(cls, dictionary)
        p.ParseList('operating_systems', func=lambda value: platforms.OperatingSystem.FromId(value, error_on_unknown=False))
        p.ParseList('architectures', func=lambda value: platforms.Architecture.FromId(value, error_on_unknown=False))
        return cls(**p.Args())

    def ToDictionary(self):
        w = DictionaryWriter(self)
        w.WriteList('operating_systems', func=DictionaryWriter.AttributeGetter('id'))
        w.WriteList('architectures', func=DictionaryWriter.AttributeGetter('id'))
        return w.Dictionary()

    def __init__(self, operating_systems, architectures):
        """Creates a new ComponentPlatform.

    Args:
      operating_systems: list(platforms.OperatingSystem), The OSes this
        component should be installed on.  None indicates all OSes.
      architectures: list(platforms.Architecture), The processor architectures
        this component works on.  None indicates all architectures.
    """
        self.operating_systems = operating_systems and sorted(operating_systems, key=lambda x: (0, x) if x is None else (1, x))
        self.architectures = architectures and sorted(architectures, key=lambda x: (0, x) if x is None else (1, x))

    def Matches(self, platform):
        """Determines if the platform for this component matches the environment.

    For both operating system and architecture, it is a match if:
     - No filter is given (regardless of platform value)
     - A filter is given but the value in platform matches one of the values in
       the filter.

    It is a match iff both operating system and architecture match.

    Args:
      platform: platform.Platform, The platform that must be matched. None will
        match only platform-independent components.

    Returns:
      True if it matches or False if not.
    """
        if not platform:
            my_os, my_arch = (None, None)
        else:
            my_os, my_arch = (platform.operating_system, platform.architecture)
        if self.operating_systems:
            if not my_os or my_os not in self.operating_systems:
                return False
        if self.architectures:
            if not my_arch or my_arch not in self.architectures:
                return False
        return True

    def IntersectsWith(self, other):
        """Determines if this platform intersects with the other platform.

    Platforms intersect if they can both potentially be installed on the same
    system.

    Args:
      other: ComponentPlatform, The other component platform to compare against.

    Returns:
      bool, True if there is any intersection, False otherwise.
    """
        return self.__CollectionsIntersect(self.operating_systems, other.operating_systems) and self.__CollectionsIntersect(self.architectures, other.architectures)

    def __CollectionsIntersect(self, collection1, collection2):
        """Determines if the two collections intersect.

    The collections intersect if either or both are None or empty, or if they
    contain an intersection of elements.

    Args:
      collection1: [] or None, The first collection.
      collection2: [] or None, The second collection.

    Returns:
      bool, True if there is an intersection, False otherwise.
    """
        if not collection1 or not collection2:
            return True
        return set(collection1) & set(collection2)