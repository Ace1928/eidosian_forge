from __future__ import annotations
import logging
from copy import copy
import zmq
class TopicLogger(logging.Logger):
    """A simple wrapper that takes an additional argument to log methods.

    All the regular methods exist, but instead of one msg argument, two
    arguments: topic, msg are passed.

    That is::

        logger.debug('msg')

    Would become::

        logger.debug('topic.sub', 'msg')
    """

    def log(self, level, topic, msg, *args, **kwargs):
        """Log 'msg % args' with level and topic.

        To pass exception information, use the keyword argument exc_info
        with a True value::

            logger.log(level, "zmq.fun", "We have a %s",
                    "mysterious problem", exc_info=1)
        """
        logging.Logger.log(self, level, f'{topic}{TOPIC_DELIM}{msg}', *args, **kwargs)