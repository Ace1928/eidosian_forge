import copy
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from oslo_db import exception as db_exception
from keystone.common import driver_hints
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.limit.backends import base
class UnifiedLimit(base.UnifiedLimitDriverBase):

    def _check_unified_limit_unique(self, unified_limit, is_registered_limit=True):
        hints = driver_hints.Hints()
        if is_registered_limit:
            hints.add_filter('service_id', unified_limit['service_id'])
            hints.add_filter('resource_name', unified_limit['resource_name'])
            hints.add_filter('region_id', unified_limit.get('region_id'))
            with sql.session_for_read() as session:
                query = session.query(RegisteredLimitModel)
                unified_limits = sql.filter_limit_query(RegisteredLimitModel, query, hints).all()
        else:
            hints.add_filter('registered_limit_id', unified_limit['registered_limit_id'])
            is_project_limit = True if unified_limit.get('project_id') else False
            if is_project_limit:
                hints.add_filter('project_id', unified_limit['project_id'])
            else:
                hints.add_filter('domain_id', unified_limit['domain_id'])
            with sql.session_for_read() as session:
                query = session.query(LimitModel)
                unified_limits = sql.filter_limit_query(LimitModel, query, hints).all()
        if unified_limits:
            msg = _('Duplicate entry')
            limit_type = 'registered_limit' if is_registered_limit else 'limit'
            raise exception.Conflict(type=limit_type, details=msg)

    def _check_referenced_limit_reference(self, registered_limit):
        with sql.session_for_read() as session:
            limits = session.query(LimitModel).filter_by(registered_limit_id=registered_limit['id'])
        if limits.all():
            raise exception.RegisteredLimitError(id=registered_limit.id)

    @sql.handle_conflicts(conflict_type='registered_limit')
    def create_registered_limits(self, registered_limits):
        with sql.session_for_write() as session:
            new_registered_limits = []
            for registered_limit in registered_limits:
                self._check_unified_limit_unique(registered_limit)
                ref = RegisteredLimitModel.from_dict(registered_limit)
                session.add(ref)
                new_registered_limits.append(ref.to_dict())
            return new_registered_limits

    @sql.handle_conflicts(conflict_type='registered_limit')
    def update_registered_limit(self, registered_limit_id, registered_limit):
        try:
            with sql.session_for_write() as session:
                ref = self._get_registered_limit(session, registered_limit_id)
                self._check_referenced_limit_reference(ref)
                old_dict = ref.to_dict()
                old_dict.update(registered_limit)
                if registered_limit.get('service_id') or 'region_id' in registered_limit or registered_limit.get('resource_name'):
                    self._check_unified_limit_unique(old_dict)
                new_registered_limit = RegisteredLimitModel.from_dict(old_dict)
                for attr in registered_limit:
                    if attr != 'id':
                        setattr(ref, attr, getattr(new_registered_limit, attr))
                return ref.to_dict()
        except db_exception.DBReferenceError:
            raise exception.RegisteredLimitError(id=registered_limit_id)

    @driver_hints.truncated
    def list_registered_limits(self, hints):
        with sql.session_for_read() as session:
            registered_limits = session.query(RegisteredLimitModel)
            registered_limits = sql.filter_limit_query(RegisteredLimitModel, registered_limits, hints)
            return [s.to_dict() for s in registered_limits]

    def _get_registered_limit(self, session, registered_limit_id):
        query = session.query(RegisteredLimitModel).filter_by(id=registered_limit_id)
        ref = query.first()
        if ref is None:
            raise exception.RegisteredLimitNotFound(id=registered_limit_id)
        return ref

    def get_registered_limit(self, registered_limit_id):
        with sql.session_for_read() as session:
            return self._get_registered_limit(session, registered_limit_id).to_dict()

    def delete_registered_limit(self, registered_limit_id):
        try:
            with sql.session_for_write() as session:
                ref = self._get_registered_limit(session, registered_limit_id)
                self._check_referenced_limit_reference(ref)
                session.delete(ref)
        except db_exception.DBReferenceError:
            raise exception.RegisteredLimitError(id=registered_limit_id)

    def _check_and_fill_registered_limit_id(self, limit):
        hints = driver_hints.Hints()
        limit_copy = copy.deepcopy(limit)
        hints.add_filter('service_id', limit_copy.pop('service_id'))
        hints.add_filter('resource_name', limit_copy.pop('resource_name'))
        hints.add_filter('region_id', limit_copy.pop('region_id', None))
        with sql.session_for_read() as session:
            registered_limits = session.query(RegisteredLimitModel)
            registered_limits = sql.filter_limit_query(RegisteredLimitModel, registered_limits, hints)
        reg_limits = registered_limits.all()
        if not reg_limits:
            raise exception.NoLimitReference
        limit_copy['registered_limit_id'] = reg_limits[0]['id']
        return limit_copy

    @sql.handle_conflicts(conflict_type='limit')
    def create_limits(self, limits):
        try:
            with sql.session_for_write() as session:
                new_limits = []
                for limit in limits:
                    target = self._check_and_fill_registered_limit_id(limit)
                    self._check_unified_limit_unique(target, is_registered_limit=False)
                    ref = LimitModel.from_dict(target)
                    session.add(ref)
                    new_limit = ref.to_dict()
                    new_limit['service_id'] = limit['service_id']
                    new_limit['region_id'] = limit.get('region_id')
                    new_limit['resource_name'] = limit['resource_name']
                    new_limits.append(new_limit)
                return new_limits
        except db_exception.DBReferenceError:
            raise exception.NoLimitReference()

    @sql.handle_conflicts(conflict_type='limit')
    def update_limit(self, limit_id, limit):
        with sql.session_for_write() as session:
            ref = self._get_limit(session, limit_id)
            if limit.get('resource_limit'):
                ref.resource_limit = limit['resource_limit']
            if limit.get('description'):
                ref.description = limit['description']
            return ref.to_dict()

    @driver_hints.truncated
    def list_limits(self, hints):
        with sql.session_for_read() as session:
            query = session.query(LimitModel).outerjoin(RegisteredLimitModel)
            limits = sql.filter_limit_query(LimitModel, query, hints)
            return [limit.to_dict() for limit in limits]

    def _get_limit(self, session, limit_id):
        query = session.query(LimitModel).filter_by(id=limit_id)
        ref = query.first()
        if ref is None:
            raise exception.LimitNotFound(id=limit_id)
        return ref

    def get_limit(self, limit_id):
        with sql.session_for_read() as session:
            return self._get_limit(session, limit_id).to_dict()

    def delete_limit(self, limit_id):
        with sql.session_for_write() as session:
            ref = self._get_limit(session, limit_id)
            session.delete(ref)

    def delete_limits_for_project(self, project_id):
        limit_ids = []
        with sql.session_for_write() as session:
            query = session.query(LimitModel)
            query = query.filter_by(project_id=project_id)
            for limit in query.all():
                limit_ids.append(limit.id)
            query.delete()
        return limit_ids