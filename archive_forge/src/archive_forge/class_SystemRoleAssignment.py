from keystone.assignment.backends import base
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
class SystemRoleAssignment(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'system_assignment'
    attributes = ['type', 'actor_id', 'target_id', 'role_id', 'inherited']
    type = sql.Column(sql.String(64), nullable=False)
    actor_id = sql.Column(sql.String(64), nullable=False)
    target_id = sql.Column(sql.String(64), nullable=False)
    role_id = sql.Column(sql.String(64), nullable=False)
    inherited = sql.Column(sql.Boolean, default=False, nullable=False)
    __table_args__ = (sql.PrimaryKeyConstraint('type', 'actor_id', 'target_id', 'role_id', 'inherited'), sql.Index('ix_system_actor_id', 'actor_id'))

    def to_dict(self):
        """Override parent method with a simpler implementation.

        RoleAssignment doesn't have non-indexed 'extra' attributes, so the
        parent implementation is not applicable.
        """
        return dict(self.items())