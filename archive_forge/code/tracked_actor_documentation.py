from typing import Callable, Optional
Actor tracked by an actor manager.

    This object is used to reference a Ray actor on an actor manager

    Existence of this object does not mean that the Ray actor has already been started.
    Actor state can be inquired from the actor manager tracking the Ray actor.

    Note:
        Objects of this class are returned by the :class:`RayActorManager`.
        This class should not be instantiated manually.

    Attributes:
        actor_id: ID for identification of the actor within the actor manager. This
            ID is not related to the Ray actor ID.

    