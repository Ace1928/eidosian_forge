from typing import Optional
class MailboxCollision(MailboxException):

    def __str__(self) -> str:
        return 'Mailbox named %s already exists' % self.args