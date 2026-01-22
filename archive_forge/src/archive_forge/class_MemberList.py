from decimal import Decimal
from boto.compat import filter, map
class MemberList(Element):

    def __init__(self, _member=None, _hint=None, *args, **kw):
        message = 'Invalid `member` specification in {0}'.format(self.__class__.__name__)
        assert 'member' not in kw, message
        if _member is None:
            if _hint is None:
                super(MemberList, self).__init__(*args, member=ElementList(**kw))
            else:
                super(MemberList, self).__init__(_hint=_hint)
        elif _hint is None:
            if issubclass(_member, DeclarativeType):
                member = _member(**kw)
            else:
                member = ElementList(_member, **kw)
            super(MemberList, self).__init__(*args, member=member)
        else:
            message = 'Nonsensical {0} hint {1!r}'.format(self.__class__.__name__, _hint)
            raise AssertionError(message)

    def teardown(self, *args, **kw):
        if self._value is None:
            self._value = []
        else:
            if isinstance(self._value.member, DeclarativeType):
                self._value.member = []
            self._value = self._value.member
        super(MemberList, self).teardown(*args, **kw)