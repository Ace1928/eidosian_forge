import copy
from sqlalchemy import inspect
from sqlalchemy import orm
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import utils
def update_on_match(query, specimen, surrogate_key, values=None, attempts=3, include_only=None, process_query=None, handle_failure=None):
    """Emit an UPDATE statement matching the given specimen.

    E.g.::

        with enginefacade.writer() as session:
            specimen = MyInstance(
                uuid='ccea54f',
                interface_id='ad33fea',
                vm_state='SOME_VM_STATE',
            )

            values = {
                'vm_state': 'SOME_NEW_VM_STATE'
            }

            base_query = model_query(
                context, models.Instance,
                project_only=True, session=session)

            hostname_query = model_query(
                    context, models.Instance, session=session,
                    read_deleted='no').
                filter(func.lower(models.Instance.hostname) == 'SOMEHOSTNAME')

            surrogate_key = ('uuid', )

            def process_query(query):
                return query.where(~exists(hostname_query))

            def handle_failure(query):
                try:
                    instance = base_query.one()
                except NoResultFound:
                    raise exception.InstanceNotFound(instance_id=instance_uuid)

                if session.query(hostname_query.exists()).scalar():
                    raise exception.InstanceExists(
                        name=values['hostname'].lower())

                # try again
                return False

            persistent_instance = base_query.update_on_match(
                specimen,
                surrogate_key,
                values=values,
                process_query=process_query,
                handle_failure=handle_failure
            )

    The UPDATE statement is constructed against the given specimen
    using those values which are present to construct a WHERE clause.
    If the specimen contains additional values to be ignored, the
    ``include_only`` parameter may be passed which indicates a sequence
    of attributes to use when constructing the WHERE.

    The UPDATE is performed against an ORM Query, which is created from
    the given ``Session``, or alternatively by passing the ```query``
    parameter referring to an existing query.

    Before the query is invoked, it is also passed through the callable
    sent as ``process_query``, if present.  This hook allows additional
    criteria to be added to the query after it is created but before
    invocation.

    The function will then invoke the UPDATE statement and check for
    "success" one or more times, up to a maximum of that passed as
    ``attempts``.

    The initial check for "success" from the UPDATE statement is that the
    number of rows returned matches 1.  If zero rows are matched, then
    the UPDATE statement is assumed to have "failed", and the failure handling
    phase begins.

    The failure handling phase involves invoking the given ``handle_failure``
    function, if any.  This handler can perform additional queries to attempt
    to figure out why the UPDATE didn't match any rows.  The handler,
    upon detection of the exact failure condition, should throw an exception
    to exit; if it doesn't, it has the option of returning True or False,
    where False means the error was not handled, and True means that there
    was not in fact an error, and the function should return successfully.

    If the failure handler is not present, or returns False after ``attempts``
    number of attempts, then the function overall raises CantUpdateException.
    If the handler returns True, then the function returns with no error.

    The return value of the function is a persistent version of the given
    specimen; this may be the specimen itself, if no matching object were
    already present in the session; otherwise, the existing object is
    returned, with the state of the specimen merged into it.  The returned
    persistent object will have the given values populated into the object.

    The object is is returned as "persistent", meaning that it is
    associated with the given
    Session and has an identity key (that is, a real primary key
    value).

    In order to produce this identity key, a strategy must be used to
    determine it as efficiently and safely as possible:

    1. If the given specimen already contained its primary key attributes
       fully populated, then these attributes were used as criteria in the
       UPDATE, so we have the primary key value; it is populated directly.

    2. If the target backend supports RETURNING, then when the update() query
       is performed with a RETURNING clause so that the matching primary key
       is returned atomically.  This currently includes Postgresql, Oracle
       and others (notably not MySQL or SQLite).

    3. If the target backend is MySQL, and the given model uses a
       single-column, AUTO_INCREMENT integer primary key value (as is
       the case for Nova), MySQL's recommended approach of making use
       of ``LAST_INSERT_ID(expr)`` is used to atomically acquire the
       matching primary key value within the scope of the UPDATE
       statement, then it fetched immediately following by using
       ``SELECT LAST_INSERT_ID()``.
       http://dev.mysql.com/doc/refman/5.0/en/information-       functions.html#function_last-insert-id

    4. Otherwise, for composite keys on MySQL or other backends such
       as SQLite, the row as UPDATED must be re-fetched in order to
       acquire the primary key value.  The ``surrogate_key``
       parameter is used for this in order to re-fetch the row; this
       is a column name with a known, unique value where
       the object can be fetched.


    """
    if values is None:
        values = {}
    entity = inspect(specimen)
    mapper = entity.mapper
    if [desc['type'] for desc in query.column_descriptions] != [mapper.class_]:
        raise AssertionError('Query does not match given specimen')
    criteria = manufacture_entity_criteria(specimen, include_only=include_only, exclude=[surrogate_key])
    query = query.filter(criteria)
    if process_query:
        query = process_query(query)
    surrogate_key_arg = (surrogate_key, entity.attrs[surrogate_key].loaded_value)
    pk_value = None
    for attempt in range(attempts):
        try:
            pk_value = query.update_returning_pk(values, surrogate_key_arg)
        except MultiRowsMatched:
            raise
        except NoRowsMatched:
            if handle_failure and handle_failure(query):
                break
        else:
            break
    else:
        raise NoRowsMatched('Zero rows matched for %d attempts' % attempts)
    if pk_value is None:
        pk_value = entity.mapper.primary_key_from_instance(specimen)
    values = copy.copy(values)
    values[surrogate_key] = surrogate_key_arg[1]
    persistent_obj = manufacture_persistent_object(query.session, specimen.__class__(), values, pk_value)
    return persistent_obj