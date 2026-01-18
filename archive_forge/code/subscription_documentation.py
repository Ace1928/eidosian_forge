from zaqarclient.queues.v2 import core
Ensures subscription exists

        This method is not race safe, the subscription could've been deleted
        right after it was called.
        