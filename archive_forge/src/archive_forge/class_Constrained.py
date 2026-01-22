from twisted.python import reflect
class Constrained(Collection):
    """A collection that has constraints on its names and/or entities."""

    def nameConstraint(self, name):
        """A method that determines whether an entity may be added to me with a given name.

        If the constraint is satisfied, return 1; if the constraint is not
        satisfied, either return 0 or raise a descriptive ConstraintViolation.
        """
        return 1

    def entityConstraint(self, entity):
        """A method that determines whether an entity may be added to me.

        If the constraint is satisfied, return 1; if the constraint is not
        satisfied, either return 0 or raise a descriptive ConstraintViolation.
        """
        return 1

    def reallyPutEntity(self, name, entity):
        Collection.putEntity(self, name, entity)

    def putEntity(self, name, entity):
        """Store an entity if it meets both constraints.

        Otherwise raise a ConstraintViolation.
        """
        if self.nameConstraint(name):
            if self.entityConstraint(entity):
                self.reallyPutEntity(name, entity)
            else:
                raise ConstraintViolation('Entity constraint violated.')
        else:
            raise ConstraintViolation('Name constraint violated.')