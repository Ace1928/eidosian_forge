from __future__ import absolute_import
import os
class BackendEntry(validation.Validated):
    """A backend entry describes a single backend."""
    ATTRIBUTES = {NAME: NAME_REGEX, CLASS: validation.Optional(CLASS_REGEX), INSTANCES: validation.Optional(validation.TYPE_INT), MAX_CONCURRENT_REQUESTS: validation.Optional(validation.TYPE_INT), OPTIONS: validation.Optional(OPTIONS_REGEX), PUBLIC: validation.Optional(validation.TYPE_BOOL), DYNAMIC: validation.Optional(validation.TYPE_BOOL), FAILFAST: validation.Optional(validation.TYPE_BOOL), START: validation.Optional(FILE_REGEX), STATE: validation.Optional(STATE_REGEX)}

    def __init__(self, *args, **kwargs):
        super(BackendEntry, self).__init__(*args, **kwargs)
        self.Init()

    def Init(self):
        if self.public:
            raise BadConfig("Illegal field: 'public'")
        if self.dynamic:
            raise BadConfig("Illegal field: 'dynamic'")
        if self.failfast:
            raise BadConfig("Illegal field: 'failfast'")
        self.ParseOptions()
        return self

    def set_class(self, Class):
        """Setter for 'class', since an attribute reference is an error."""
        self.Set(CLASS, Class)

    def get_class(self):
        """Accessor for 'class', since an attribute reference is an error."""
        return self.Get(CLASS)

    def ToDict(self):
        """Returns a sorted dictionary representing the backend entry."""
        self.ParseOptions().WriteOptions()
        result = super(BackendEntry, self).ToDict()
        return validation.SortedDict([NAME, CLASS, INSTANCES, START, OPTIONS, MAX_CONCURRENT_REQUESTS, STATE], result)

    def ParseOptions(self):
        """Parses the 'options' field and sets appropriate fields."""
        if self.options:
            options = [option.strip() for option in self.options.split(',')]
        else:
            options = []
        for option in options:
            if option not in VALID_OPTIONS:
                raise BadConfig('Unrecognized option: %s', option)
        self.public = PUBLIC in options
        self.dynamic = DYNAMIC in options
        self.failfast = FAILFAST in options
        return self

    def WriteOptions(self):
        """Writes the 'options' field based on other settings."""
        options = []
        if self.public:
            options.append('public')
        if self.dynamic:
            options.append('dynamic')
        if self.failfast:
            options.append('failfast')
        if options:
            self.options = ', '.join(options)
        else:
            self.options = None
        return self