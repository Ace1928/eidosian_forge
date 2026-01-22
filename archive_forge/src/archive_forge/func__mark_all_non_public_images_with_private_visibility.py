from sqlalchemy import MetaData, select, Table, and_, not_
def _mark_all_non_public_images_with_private_visibility(engine, images):
    with engine.connect() as conn:
        migrated_rows = conn.execute(images.update().values(visibility='private').where(not_(images.c.is_public)))
    return migrated_rows.rowcount