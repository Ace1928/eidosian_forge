import inspect
import sys
class UnknownMemberIdError(BrokerResponseError):
    errno = 25
    message = 'UNKNOWN_MEMBER_ID'
    description = 'Returned from group requests (offset commits/fetches, heartbeats, etc) when the memberId is not in the current generation.'