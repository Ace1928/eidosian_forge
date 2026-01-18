import random
from parlai.core.agents import Agent
from parlai.core.message import Message

Simple agent which repeats back the labels sent to it.

By default, replies with a single random label from the list of labels sent to
it, if any. If the ``label_candidates`` field is set, will fill the ``text_candidates``
field with up to a hundred randomly selected candidates (the first text
candidate is the selected label).

Options:

    ``returnOneRandomAnswer`` -- default ``True``, set to ``False`` to instead
    reply with all labels joined by commas.

    ``cantAnswerPercent`` -- default ``0``, set value in range[0,1] to set
    chance of replying with "I don't know."
